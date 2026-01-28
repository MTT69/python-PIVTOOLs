# Image Loading Architecture

This document explains how PIVTools handles image loading for both PIV processing and calibration workflows.

---

## Architecture Overview

```
pivtools_core/image_handling/
├── load_images.py              # Main PIV image loading (read_pair, load_images)
├── calibration_loader.py       # Calibration image loading (read_calibration_image)
├── path_utils.py               # Shared path building and validation
└── readers/
    ├── __init__.py             # Reader registry (get_reader, register_reader)
    ├── lavision_reader.py      # .im7, .set readers
    ├── cine_reader.py          # .cine reader
    └── generic_readers.py      # .tif, .png, .jpg readers
```

---

## Supported Image Formats

| Format | Extension | Description | Container? |
|--------|-----------|-------------|------------|
| LaVision SET | `.set` | Multi-camera, multi-frame container | Yes |
| LaVision IM7 | `.im7` | Per-timestamp file, may contain multiple cameras | Depends* |
| Phantom CINE | `.cine` | Single-camera video container | Yes |
| TIFF | `.tif`, `.tiff` | Standard image format | No |
| PNG | `.png` | Standard image format | No |
| JPEG | `.jpg`, `.jpeg` | Standard image format | No |
| RAW | `.raw`, `.cr2`, `.nef`, `.arw` | Camera RAW formats | No |

*IM7 files with `%` pattern (e.g., `B%05d.im7`) are individual files per timestamp, NOT containers.

---

## Reader Registry

All image reading goes through a centralized registry in `readers/__init__.py`:

```python
# Registration
_READERS = {
    '.tif': read_tiff,
    '.tiff': read_tiff,
    '.png': read_png_jpeg,
    '.jpg': read_png_jpeg,
    '.im7': read_lavision_pair,
    '.set': read_lavision_ims_pair,
    '.cine': read_cine_pair,
    # ...
}

# Usage
reader_func = get_reader(file_path)  # Returns appropriate reader
image = reader_func(file_path, **kwargs)
```

---

## PIV Image Loading

### Entry Points

| Function | Location | Purpose |
|----------|----------|---------|
| `load_images()` | `load_images.py` | Load all PIV pairs for a camera (Dask lazy) |
| `read_pair()` | `load_images.py` | Read single PIV pair (A + B frames) |
| `read_single_frame()` | `load_images.py` | Read one frame from any format |

### Flow Diagram

```
load_images(camera, config)
    │
    ├── build_piv_camera_path(config, camera)    [path_utils.py]
    │       → Returns: Path to camera directory or container file
    │
    └── for each frame_pair:
            delayed_image_pair(idx, camera_path, camera, config)
                │
                └── read_pair(idx, camera_path, camera, config)
                        │
                        ├── config.get_frame_pair_indices(idx)
                        │       → Returns: (frame_a_idx, frame_b_idx)
                        │
                        └── read_single_frame(file_path, camera, frame_idx, image_type)
                                │
                                └── get_reader(file_path)  [readers/__init__.py]
                                        │
                                        └── reader_func(file_path, **kwargs)
                                                → Returns: np.ndarray (H, W)
```

### Path Building for PIV

`build_piv_camera_path()` determines where to find images:

| Image Type | Path Structure |
|------------|----------------|
| `lavision_set` | `source_path` IS the `.set` file |
| `cine` | `source_path` directory (files named by camera) |
| `lavision_im7` (no subfolders) | `source_path` directory |
| `lavision_im7` (with subfolders) | `source_path / Cam{N}/` |
| Standard formats | `source_path / Cam{N}/` |

### Frame Pair Logic

The `config.get_frame_pair_indices(idx)` method handles:

- **Time-resolved**: Frame A at index `2*idx-1`, Frame B at `2*idx`
- **Non-time-resolved**: Both frames at same index (A+B stored together)
- **Zero-based indexing**: Adjusts indices if configured
- **Skip modes**: Handles non-sequential frame pairing

### Output Shape

`read_pair()` always returns: `np.ndarray` of shape `(2, H, W)`
- Index 0: Frame A
- Index 1: Frame B

---

## Calibration Image Loading

### Entry Points

| Function | Location | Purpose |
|----------|----------|---------|
| `read_calibration_image()` | `calibration_loader.py` | Read single calibration frame |
| `validate_calibration_images()` | `calibration_loader.py` | Validate images exist |
| `get_calibration_frame_count()` | `calibration_loader.py` | Auto-detect image count |

### Flow Diagram

```
read_calibration_image(idx, camera, config)
    │
    ├── build_calibration_camera_path(config, camera)    [path_utils.py]
    │       → Returns: Path to calibration directory or container
    │
    ├── resolve_file_path(camera_path, camera, idx, format, image_type)
    │       → Returns: Full path to specific image file
    │
    └── read_single_frame(file_path, camera, idx, image_type)
            │
            └── get_reader(file_path)
                    │
                    └── reader_func(file_path, **kwargs)
                            │
                            └── _normalize_to_uint8(img)  [if normalize_uint8=True]
                                    → Returns: np.ndarray (H, W) as uint8
```

### Path Building for Calibration

`build_calibration_camera_path()` uses `calibration_sources` from config:

| Image Type | use_camera_subfolders | Path Structure |
|------------|----------------------|----------------|
| `lavision_set` | N/A | `calibration_source` (path to `.set` file) |
| `cine` | N/A | `calibration_source` directory |
| `lavision_im7` | `False` | `calibration_source` directory |
| `lavision_im7` | `True` | `calibration_source / Cam{N}/` |
| Standard | `True` | `calibration_source / Cam{N}/` |
| Standard | `False` | `calibration_source` directory |

### Output Normalization

Calibration images are normalized to `uint8` by default for OpenCV compatibility:

```python
def _normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        # Min-max normalization to 0-255
        ...
```

### Output Shape

`read_calibration_image()` always returns: `np.ndarray` of shape `(H, W)` as `uint8`

---

## Container Format Detection

The `config.is_container_format` and `config.calibration_is_container_format` properties determine if a format stores multiple frames in a single file:

```python
@property
def calibration_is_container_format(self) -> bool:
    image_type = self.calibration_image_type
    image_format = self.calibration_image_format

    # IM7 files with % pattern are individual numbered files, NOT containers
    if image_type == "lavision_im7" and "%" in image_format:
        return False

    # Only .set and .cine are true multi-frame containers
    return image_type in ("cine", "lavision_set")
```

**Key distinction:**
- `B%05d.im7` → Individual files (one per timestamp) → NOT a container
- `data.set` → All frames in one file → IS a container
- `Camera1.cine` → All frames in one video → IS a container

---

## Configuration Properties

### PIV Settings (in `config.yaml`)

```yaml
paths:
  source_paths:
    - /path/to/piv/images    # or /path/to/file.set for containers
  camera_subfolders:
    - Cam1
    - Cam2

processing:
  image_type: standard       # standard | lavision_set | lavision_im7 | cine
  image_format: "%05d.tif"   # printf-style pattern
  num_frame_pairs: 100
  time_resolved: true
```

### Calibration Settings (in `config.yaml`)

```yaml
calibration:
  calibration_sources:
    - /path/to/calibration/images
  image_type: standard       # standard | lavision_set | lavision_im7 | cine
  image_format: "cal_%03d.tif"
  num_images: 20
  use_camera_subfolders: true
  zero_based_indexing: false
```

---

## Common Patterns

### Reading a PIV Pair Manually

```python
from pivtools_core.config import get_config
from pivtools_core.image_handling.load_images import read_pair
from pivtools_core.image_handling.path_utils import build_piv_camera_path

config = get_config()
camera_path = build_piv_camera_path(config, source_path_idx=0, camera=1)
pair = read_pair(idx=1, camera_path=camera_path, camera=1, config=config)
# pair.shape = (2, H, W)
```

### Reading a Calibration Image Manually

```python
from pivtools_core.config import get_config
from pivtools_core.image_handling.calibration_loader import read_calibration_image

config = get_config()
img = read_calibration_image(idx=1, camera=1, config=config)
# img.shape = (H, W), dtype = uint8
```

### Lazy Loading with Dask

```python
from pivtools_core.config import get_config
from pivtools_core.image_handling.load_images import load_images

config = get_config()
images = load_images(camera=1, config=config)
# images is a Dask array of shape (num_pairs, 2, H, W)
# No data loaded until .compute() is called
```

---

## Error Handling

### FileNotFoundError

Raised when:
- Image file doesn't exist at resolved path
- Container file (.set, .cine) not found
- Camera subfolder doesn't exist

### ValueError

Raised when:
- No reader registered for file extension
- `calibration_sources` not configured
- Invalid image format pattern

---

## File Format Details

### LaVision .set Files

- Contains all cameras and all time instances in one file
- Access via: `camera_no` (1-based) and `im_no` (frame index)
- Reader: `read_lavision_ims(file_path, camera_no=1, im_no=1)`

### LaVision .im7 Files

- One file per timestamp (when using `%` pattern)
- May contain multiple cameras per file
- Access via: `camera_no` (1-based), `frames`, `frames_per_camera`
- Reader: `read_lavision_im7(file_path, camera_no=1, frames=1)`

### Phantom .cine Files

- One video file per camera
- File naming uses camera number: `Camera%d.cine`
- Access via: `idx` (frame index), `frames` (how many to read)
- Reader: `read_cine_pair(file_path, idx=1, frames=2)`

### Standard Formats (.tif, .png, .jpg)

- One file per frame per camera
- Organized in camera subfolders: `Cam1/00001.tif`
- File naming uses frame index: `%05d.tif`
- Reader: `read_tiff(file_path)` or `read_png_jpeg(file_path)`
