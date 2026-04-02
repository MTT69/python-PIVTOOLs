# Custom LaVision File Readers — Migration from lvpyio

## Why

`lvpyio` is a C++ wrapper (Windows-only) for reading LaVision's proprietary `.im7` and `.set` image formats. It had three fundamental problems:

1. **Platform lock-in** — would not install on macOS, blocking development on Mac entirely
2. **Reads everything** — when you request camera 4 from an 8-frame .im7 file, lvpyio decodes all 8 frames into float64 arrays via its C++ layer, then the Python wrapper iterates a generator to skip the ones you don't want. The data is already in memory before your code touches it.
3. **Float64 intermediates** — pixel data (uint16 on disk) is promoted to float64 during scale application, creating arrays 4x larger than needed

## What We Built

Two pure-Python readers that replace lvpyio entirely:

- **`im7_reader.py`** — reads LaVision `.im7` files (256-byte binary header + pixel data + attributes)
- **`set_reader.py`** — reads LaVision `.set` containers (index files + 20 GB data files + mono-12p pixel decoding)

Both live in `pivtools_core/image_handling/readers/` and are used by the existing `lavision_reader.py` wrappers (same public API, zero downstream changes).

## Benchmark Results

Tested on real PIV data (4-camera, 3248x4872 uint16 .im7 and 2-camera, 3352x5312 mono-12p .set).

### .im7 — Multi-Camera PIV (reading camera 4, skipping 6 frames)

| Metric | lvpyio | Custom | Factor |
|--------|--------|--------|--------|
| Peak RAM | 886 MB | 158 MB | **5.6x less** |
| Speed | 514 ms | 58 ms | **8.8x faster** |

### .im7 — Single-Frame Calibration Image (5312x4600 float32)

| Metric | lvpyio | Custom | Factor |
|--------|--------|--------|--------|
| Peak RAM | 513 MB | 196 MB | **2.6x less** |
| Speed | 339 ms | 57 ms | **5.9x faster** |

### .set — Pre-Paired PIV (camera 2, entry 500 in a 20 GB file)

| Metric | lvpyio | Custom | Factor |
|--------|--------|--------|--------|
| Peak RAM | 855 MB | 312 MB | **2.7x less** |
| Speed | 449 ms | 195 ms | **2.3x faster** |

All results are **bit-identical** (`np.array_equal = True`) across 19 test cases covering single-frame, multi-frame, and per-camera extraction.

## How It Works

### .im7 Format

```
[256-byte header] → version, pack_type, buffer_format, sizeX/Y/Z/F
[pixel data]      → contiguous frames, each sizeZ * sizeY * sizeX pixels
[attributes]      → scale records (slope, offset, unit)
```

**Pack types supported:**
- `0` — Uncompressed (float32 or uint16). Direct seek to target frame offset.
- `2` — Zlib row-compressed. Each row: 4-byte compressed size + zlib payload. Skip by reading size prefix and seeking past payload.
- `3` — Fixed 12-bit packed. 4 pixels in 3 uint16 words. Seek by calculated byte offset.

**Frame skipping:** For uncompressed/12-bit, a single `f.seek()` jumps past unwanted frames (zero I/O cost). For zlib, the reader reads 4-byte size prefixes and seeks past compressed payloads without decompressing.

### .set Format

```
recording.set           ← tiny marker file (0 bytes of data)
recording/              ← companion directory
  StreamSet.xml         ← declares frame streams
  Frame{N}-0.ims        ← index: 256-byte header + 768-byte padding + entries * 20 bytes
  Frame{N}-1.ims        ← data: raw packed pixels (e.g., 20 GB)
  Frame{N}-decoder.xml  ← pixel format ("mono-12p", "mono-16")
  FrameScales{N}.scales ← XML intensity scales
```

**Index entry:** 20 bytes = int32 flag + int64 offset + int64 size. Gives exact byte position in the data file for each image.

**Mono12Packed:** GigE Vision standard. 2 pixels in 3 bytes:
```
pixel0 = byte0 | (byte1 & 0x0F) << 8
pixel1 = (byte1 >> 4) | (byte2 << 4)
```

**Direct access:** `read_set_pair(path, camera_no=2, im_no=500)` parses the index, seeks to byte offset `500 * 26,708,736` in the 20 GB file, reads exactly 26.7 MB, decodes, done. No scanning or iteration.

## Key Optimisations

1. **Frame skipping via seek** — `read_im7_camera()` only reads the frames for the requested camera. For uncompressed data, `f.seek()` jumps past unwanted frames without touching memory.

2. **Float32 in-place scale** — `result = pixels.astype(np.float32); result *= slope; result += offset`. Avoids numpy's default uint16 * float → float64 promotion. Saves 4x memory on scale application.

3. **`del frame, raw` to break view chains** — `np.frombuffer(raw)` creates a view that keeps the bytes object alive. Explicit `del` frees each frame's raw data before the next `f.read()`, preventing two frames from coexisting in memory.

4. **`set_info` caching** — `read_set_info()` parses index files once; pass the result to `read_set_pair(..., set_info=info)` for batch reads. Eliminates ~200 ms/call of index re-parsing (halves per-pair time for 750-pair batches).

5. **Frame-by-frame output construction** — Pre-allocate the float32 output array, read one raw frame at a time, cast directly into the output slice. Peak memory = output + one frame's raw data (not all frames at once).

## Validation

19 test cases verified bit-exact (`np.array_equal`) against lvpyio:
- 6 single-frame calibration images (camera1, float32 uncompressed)
- 1 multi-frame PIV file (8 frames, uint16 uncompressed, full read)
- 4 per-camera extractions (read_im7_camera vs read_lavision_im7)
- 8 .set pair reads (2 cameras x 4 entries: 1, 5, 100, 750)

Validation script: `lavision_conversion/compare_readers.py` (requires lvpyio installed for comparison).

## What's Untested

These paths are implemented from the LaVision spec but have no real test data:
- Zlib-compressed .im7 (`pack_type=2`) — row-by-row decompression logic
- IM7 12-bit packed (`pack_type=3`) — bit-shift unpacking from uint16 words
- .set with `mono-16` decoder — trivial `np.frombuffer(raw, uint16)` path
- Non-trivial intensity scales (slope != 1.0 or offset != 0.0)
- `sizeZ > 1` in .im7 (multi-plane, rare in PIV)

All untested paths will either work correctly or raise a clear error — no silent corruption possible.
