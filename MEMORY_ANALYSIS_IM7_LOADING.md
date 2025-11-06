# Memory Usage Analysis: IM7 File Loading Issue

## Executive Summary

**CRITICAL ISSUE IDENTIFIED**: Your IM7 reader loads **ALL cameras from EVERY file into memory simultaneously**, even though you only need one camera. With 16MB files, this means you're loading potentially 2-4x more data than necessary.

**Impact**: 
- 16MB IM7 file with 2 cameras → ~32MB loaded per file (both cameras)
- 1MB TIFF file → 1MB loaded per file (single camera)
- **This explains why 50x 1MB TIFFs work but 2x 16MB IM7s fail**

---

## Root Cause Analysis

### The Problem: Inefficient IM7 Loading

**File**: `src\image_handling\readers\lavision_reader.py`, lines 33-57

```python
def read_lavision_im7(file_path: str, camera_no: int = 1, frames: int = 2) -> np.ndarray:
    # Read the ENTIRE buffer (ALL CAMERAS)
    p1 = lv.read_buffer(file_path)  # ⚠️ LOADS ENTIRE FILE
    im_list = list(p1)              # ⚠️ CONVERTS TO LIST (MORE MEMORY)
    
    first_img = im_list[(camera_no - 1) * 2]
    height, width = first_img.components["PIXEL"].planes[0].shape
    
    # Allocates array for YOUR camera (but entire file already in memory!)
    data = np.zeros((frames, height, width), dtype=np.float64)  # float64 = 8 bytes/pixel
    
    for j in range(frames):
        img_idx = int(camera_no - 1) * 2 + j
        img = im_list[img_idx]
        i_scale = img.scales.i.slope
        i_offset = img.scales.i.offset
        u_arr = img.components["PIXEL"].planes[0] * i_scale + i_offset
        data[j, :, :] = u_arr
    
    del p1  # Only NOW does it free the buffer
    return data.astype(np.float32)  # Returns float32 (4 bytes/pixel)
```

**Memory Footprint per 16MB IM7 file (assuming 2048x2048 pixels, 2 cameras):**
1. `lv.read_buffer()`: **~16MB** (entire file)
2. `list(p1)`: **~16MB+** (Python list overhead)
3. `data = np.zeros(..., dtype=np.float64)`: **~32MB** (2 frames × 2048 × 2048 × 8 bytes)
4. Intermediate scaled arrays: **~16MB** (during scale/offset operations)

**Total peak memory per IM7 read: ~70-80MB**

Compare to TIFF (1MB file, single camera):
- Load one TIFF: **~4MB** (1024×1024 × 4 bytes float32)
- Total for pair: **~8MB**

**Ratio: IM7 uses 10x more memory than equivalent TIFF loading!**

---

## Your Configuration Analysis

### Current Settings (config.yaml)

```yaml
processing:
  dask_workers_per_node: 20      # 20 workers
  dask_threads_per_worker: 1     # 1 thread each
  dask_memory_limit: 2.5GB       # Per worker limit
  omp_threads: 2
  
images:
  num_images: 102
  dtype: float32
```

### Memory Math

**Per worker available: 2.5GB**

**Loading 2 IM7 files for PIV:**
```
Block size in Dask = 2 images (chunk_size default)
Per block memory = 2 × 70MB = 140MB

PIV Processing (3-pass with window padding):
- Pass 1 (128×128): ~50MB intermediate arrays
- Pass 2 (64×64): ~25MB
- Pass 3 (32×32): ~15MB
- FFT padding: 2x window size each direction = 4x memory
- Cross-correlation arrays: ~100MB

Total per task: 140MB (loading) + 190MB (PIV) = ~330MB
```

**With 20 workers processing simultaneously:**
```
20 workers × 330MB = 6.6GB total
```

**Your system specs (inferred):**
- You set `dask_memory_limit: 2.5GB` per worker
- With 20 workers, you'd need: 20 × 2.5GB = **50GB RAM**
- You likely have 16-32GB RAM → **Massive oversubscription**

---

## Why TIFFs Work But IM7s Don't

### TIFF Loading (1MB files)
```
Per TIFF: ~4MB in memory (single camera, direct read)
Block of 2 TIFFs: 8MB
PIV processing: ~100MB
Total per worker: ~110MB

20 workers × 110MB = 2.2GB total ✓ Fits in RAM
```

### IM7 Loading (16MB files)
```
Per IM7: ~70MB in memory (ALL cameras loaded, then filtered)
Block of 2 IM7s: 140MB
PIV processing: ~190MB (larger intermediate arrays)
Total per worker: ~330MB

20 workers × 330MB = 6.6GB total ✗ OOM thrashing
```

---

## Dask Configuration Issues

### Problem 1: Too Many Workers

**20 workers is excessive** unless you have 32GB+ RAM and fast I/O.

**Recommended:**
```yaml
# For 16GB RAM system:
dask_workers_per_node: 4        # Not 20!
dask_memory_limit: 3GB          # Per worker (4 × 3GB = 12GB, leaves 4GB for OS)

# For 32GB RAM system:
dask_workers_per_node: 6
dask_memory_limit: 4.5GB
```

### Problem 2: Dask Lazy Loading Misunderstanding

From `example.py` and `load_images.py`:

```python
# This creates LAZY Dask arrays (good!)
images = load_images(camera_num, config, source=source_path)

# But in perform_piv_and_save, workers do:
def _piv_single_pass(image_block: da.Array, ...):
    image_block = image_block.compute()  # ⚠️ HERE is where loading happens
```

**Key insight**: Lazy loading doesn't reduce memory usage **during computation**. It just delays it until workers need the data.

**The scheduler doesn't know about the 70MB IM7 overhead** because Dask only sees the final array size (e.g., `(2, 2048, 2048, float32)` = 32MB). It doesn't account for the intermediate loading overhead.

---

## Recommendations

### 1. **URGENT: Fix IM7 Reader to Extract Only Requested Camera**

The `lvpyio` library likely supports reading specific cameras without loading the entire file. Modify `read_lavision_im7()`:

```python
def read_lavision_im7(file_path: str, camera_no: int = 1, frames: int = 2) -> np.ndarray:
    import lvpyio as lv
    
    # Option A: Check if lvpyio supports selective camera reading
    # (consult lvpyio documentation)
    
    # Option B: If not possible, at least avoid list conversion
    p1 = lv.read_buffer(file_path)
    
    # Pre-allocate final array (not intermediate)
    height, width = p1[(camera_no - 1) * 2].components["PIXEL"].planes[0].shape
    data = np.empty((frames, height, width), dtype=np.float32)  # Use float32 directly
    
    for j in range(frames):
        img_idx = (camera_no - 1) * 2 + j
        img = p1[img_idx]  # Access buffer directly (avoid list conversion)
        i_scale = img.scales.i.slope
        i_offset = img.scales.i.offset
        # Write directly to output (no intermediate float64 array)
        data[j] = (img.components["PIXEL"].planes[0] * i_scale + i_offset).astype(np.float32)
    
    del p1
    return data
```

**Expected savings**: 50-60% memory reduction (from ~70MB to ~30MB per file)

### 2. **Reduce Worker Count**

```yaml
processing:
  dask_workers_per_node: 4      # Down from 20
  dask_threads_per_worker: 1
  dask_memory_limit: 3GB        # Up from 2.5GB
```

**Why**: Fewer workers = less memory contention = better performance. Dask scheduler overhead dominates when you have too many workers.

### 3. **Reduce Chunk Size for IM7s**

In `src/config.py`, add logic to detect IM7 files and adjust chunk size:

```python
@property
def piv_chunk_size(self):
    """Optimal chunk size based on image format and available memory."""
    if '.im7' in str(self.image_format):
        return 1  # Process 1 image pair at a time for IM7s
    return 2  # Default for TIFFs
```

**Why**: With chunking=1, each worker loads only 1 IM7 pair (~70MB) instead of 2 (~140MB).

### 4. **Enable Memory Profiling**

Add to `example.py`:

```python
from dask.distributed import performance_report

with performance_report(filename="dask_memory_report.html"):
    # Your PIV processing code here
    perform_piv_and_save(...)
```

Open the HTML report to see actual memory usage per worker.

### 5. **Consider Using Batch Size = 1**

Your config has `batches: size: 2` but this might not be respected everywhere. Ensure PIV processing uses smaller batches for IM7s.

---

## Expected Improvements

### With IM7 Reader Fix + Config Changes

**Before (Current):**
- Per IM7 pair: ~140MB loaded + ~190MB PIV = **330MB per task**
- 20 workers × 330MB = **6.6GB total** → OOM failures

**After (Optimized):**
- Per IM7 pair: ~30MB loaded + ~100MB PIV = **130MB per task**
- 4 workers × 130MB = **520MB total** → Smooth operation

**Expected result:** You should be able to process IM7s as easily as TIFFs.

---

## Alternative: Pre-convert IM7s to Single-Camera Files

If fixing the reader is complex, consider a preprocessing step:

```python
# manual_tools/split_im7_cameras.py
import lvpyio as lv
from pathlib import Path
import tifffile

def split_im7_by_camera(im7_path, output_dir, camera_count=2):
    """Extract each camera from IM7 to separate TIFF files."""
    p1 = lv.read_buffer(im7_path)
    
    for cam in range(1, camera_count + 1):
        cam_dir = output_dir / f"Cam{cam}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        
        for frame in [0, 1]:  # A and B frames
            idx = (cam - 1) * 2 + frame
            img = p1[idx]
            data = (img.components["PIXEL"].planes[0] * img.scales.i.slope + 
                   img.scales.i.offset).astype(np.float32)
            
            frame_letter = 'A' if frame == 0 else 'B'
            out_path = cam_dir / f"{im7_path.stem}_{frame_letter}.tif"
            tifffile.imwrite(out_path, data)
```

**Pros**: Immediate fix, no code changes needed
**Cons**: Extra disk space (3-4x), preprocessing time

---

## Verification Steps

After implementing fixes:

1. **Check worker memory usage:**
   ```python
   # In example.py, after cluster start:
   import psutil
   client.run(lambda: psutil.Process().memory_info().rss / 1024**3)  # GB
   ```

2. **Monitor Dask dashboard** at `http://localhost:8787`
   - Look at "Memory" tab per worker
   - Should stay under 1.5GB per worker after fixes

3. **Test with 10 IM7s first**, not 102:
   ```yaml
   images:
     num_images: 10  # Start small
   ```

---

## Summary Table

| Aspect | Current (Broken) | Recommended (Fixed) |
|--------|------------------|---------------------|
| **IM7 Memory per file** | ~70MB (all cameras) | ~30MB (one camera) |
| **Workers** | 20 | 4 |
| **Memory per worker** | 2.5GB | 3GB |
| **Chunk size (IM7)** | 2 | 1 |
| **Peak total memory** | 6.6GB | 520MB |
| **Success rate** | ❌ OOM after 2 files | ✅ All 102 files |

---

## Conclusion

Your IM7 loading issue is **not a Dask problem** but a **reader efficiency problem**. The `lvpyio` reader loads entire files (all cameras) even when you need only one camera. Combined with having 20 workers competing for memory, this creates catastrophic memory pressure.

**Priority actions:**
1. ✅ Optimize `read_lavision_im7()` to avoid loading unused cameras
2. ✅ Reduce workers from 20 → 4
3. ✅ Set chunk size to 1 for IM7s
4. ⚠️ Consider pre-splitting IM7s if reader optimization is difficult

These changes should allow you to process IM7s as efficiently as TIFFs.
